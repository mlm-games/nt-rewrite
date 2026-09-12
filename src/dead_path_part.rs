pub fn derive_dead_path(idle: &'static str) -> &'static str {
    let base = if idle.contains("sprMutant") {
        if idle.contains("sprMutant1BIdle") {
            return "images/sprMutant1BDead.png";
        }
        if idle.contains("sprMutant1CIdle") {
            return "images/sprMutant1CDead.png";
        }
        if idle.contains("sprMutant2BIdle") {
            return "images/sprMutant2BDead.png";
        }
        if idle.contains("sprMutant2CIdle") {
            return "images/sprMutant2CDead.png";
        }
        if idle.contains("sprMutant3BIdle") {
            return "images/sprMutant3BDead.png";
        }
        if idle.contains("sprMutant3CIdle") {
            return "images/sprMutant3CDead.png";
        }
        if idle.contains("sprMutant4BIdle") {
            return "images/sprMutant4BDead.png";
        }
        if idle.contains("sprMutant4CIdle") {
            return "images/sprMutant4CDead.png";
        }
        if idle.contains("sprMutant5BIdle") {
            return "images/sprMutant5BDead.png";
        }
        if idle.contains("sprMutant5CIdle") {
            return "images/sprMutant5CDead.png";
        }
        if idle.contains("sprMutant6BIdle") {
            return "images/sprMutant6BDead.png";
        }
        if idle.contains("sprMutant6CIdle") {
            return "images/sprMutant6CDead.png";
        }
        if idle.contains("sprMutant7BIdle") {
            return "images/sprMutant7BDead.png";
        }
        if idle.contains("sprMutant7CIdle") {
            return "images/sprMutant7CDead.png";
        }
        if idle.contains("sprMutant8BIdle") {
            return "images/sprMutant8BDead.png";
        }
        if idle.contains("sprMutant8CIdle") {
            return "images/sprMutant8CDead.png";
        }
        if idle.contains("sprMutant9BIdle") {
            return "images/sprMutant9BDead.png";
        }
        if idle.contains("sprMutant9CIdle") {
            return "images/sprMutant9CDead.png";
        }
        if idle.contains("sprMutant10BIdle") {
            return "images/sprMutant10BDead.png";
        }
        if idle.contains("sprMutant10CIdle") {
            return "images/sprMutant10CDead.png";
        }
        if idle.contains("sprMutant11BIdle") {
            return "images/sprMutant11BDead.png";
        }
        if idle.contains("sprMutant11CIdle") {
            return "images/sprMutant11CDead.png";
        }
        if idle.contains("sprMutant12BIdle") {
            return "images/sprMutant12BDead.png";
        }
        if idle.contains("sprMutant12CIdle") {
            return "images/sprMutant12CDead.png";
        }
        if idle.contains("sprMutant13BIdle") {
            return "images/sprMutant13BDead.png";
        }
        if idle.contains("sprMutant13CIdle") {
            return "images/sprMutant13CDead.png";
        }
        if idle.contains("sprMutant14BIdle") {
            return "images/sprMutant14BDead.png";
        }
        if idle.contains("sprMutant14CIdle") {
            return "images/sprMutant14CDead.png";
        }
        if idle.contains("sprMutant15BIdle") {
            return "images/sprMutant15BDead.png";
        }
        if idle.contains("sprMutant15CIdle") {
            return "images/sprMutant15CDead.png";
        }
        if idle.contains("sprMutant16BIdle") {
            return "images/sprMutant16BDead.png";
        }
        if idle.contains("sprMutant16CIdle") {
            return "images/sprMutant16CDead.png";
        }
        idle
    } else {
        idle
    };
    match base {
        "images/sprMutant1Idle.png" => "images/sprMutant1Dead.png",
        "images/sprMutant2Idle.png" => "images/sprMutant2Dead.png",
        "images/sprMutant3Idle.png" => "images/sprMutant3Dead.png",
        "images/sprMutant4Idle.png" => "images/sprMutant4Dead.png",
        "images/sprMutant5Idle.png" => "images/sprMutant5Dead.png",
        "images/sprMutant6Idle.png" => "images/sprMutant6Dead.png",
        "images/sprMutant7Idle.png" => "images/sprMutant7Dead.png",
        "images/sprMutant8Idle.png" => "images/sprMutant8Dead.png",
        "images/sprMutant9Idle.png" => "images/sprMutant9Dead.png",
        "images/sprMutant10Idle.png" => "images/sprMutant10Dead.png",
        "images/sprMutant11Idle.png" => "images/sprMutant11Dead.png",
        "images/sprMutant12Idle.png" => "images/sprMutant12Dead.png",
        "images/sprMutant13Idle.png" => "images/sprMutant13Dead.png",
        "images/sprMutant14Idle.png" => "images/sprMutant14Dead.png",
        "images/sprMutant15Idle.png" => "images/sprMutant15Dead.png",
        "images/sprMutant16Idle.png" => "images/sprMutant16Dead.png",
        "images/sprBanditIdle.png" => "images/sprBanditDead.png",
        "images/sprMaggotIdle.png" => "images/sprMaggotDead.png",
        "images/sprScorpionIdle.png" => "images/sprScorpionDead.png",
        "images/sprRatIdle.png" => "images/sprRatDead.png",
        "images/sprRatkingIdle.png" => "images/sprRatkingDead.png",
        "images/sprFreak1Idle.png" => "images/sprFreak1Dead.png",
        "images/sprJungleAssassinIdle.png" => "images/sprJungleAssassinDead.png",
        "images/sprSnowBotIdle.png" => "images/sprSnowBotDead.png",
        "images/sprTurretIdle.png" => "images/sprTurretDead.png",
        "images/sprSnowBanditIdle.png" => "images/sprSnowBanditDead.png",
        "images/sprWolfIdle.png" => "images/sprWolfDead.png",
        "images/sprBanditBossIdle.png" => "images/sprBanditBossDead.png",
        "images/sprGatorIdle.png" => "images/sprGatorDead.png",
        "images/sprBuffGatorIdle.png" => "images/sprBuffGatorDead.png",
        "images/sprRavenIdle.png" => "images/sprRavenDead.png",
        "images/sprSalamanderIdle.png" => "images/sprSalamanderDead.png",
        "images/sprMeleeIdle.png" => "images/sprMeleeDead.png",
        "images/sprJungleBanditIdle.png" => "images/sprJungleBanditDead.png",
        "images/sprBigMaggotIdle.png" => "images/sprBigMaggotDead.png",
        "images/sprFastRatIdle.png" => "images/sprFastRatDead.png",
        "images/sprGoldScorpionIdle.png" => "images/sprGoldScorpionDead.png",
        "images/sprLightningCrystalIdle.png" => "images/sprLightningCrystalDead.png",
        "images/sprExploFreakIdle.png" => "images/sprExploFreakDead.png",
        "images/sprRhinoFreakIdle.png" => "images/sprRhinoFreakDead.png",
        "images/sprSnowTankIdle.png" => "images/sprSnowTankDead.png",
        "images/sprGoldTankIdle.png" => "images/sprGoldTankDead.png",
        "images/sprGuardianIdle.png" => "images/sprGuardianDead.png",
        "images/sprExploGuardianIdle.png" => "images/sprExploGuardianDead.png",
        "images/sprDogGuardianWalk.png" => "images/sprDogGuardianDead.png",
        "images/sprBoneFish1Idle.png" => "images/sprBoneFish1Dead.png",
        "images/sprTurtleIdle.png" => "images/sprTurtleDead.png",
        "images/sprMolefishIdle.png" => "images/sprMolefishDead.png",
        "images/sprMolesargeIdle.png" => "images/sprMolesargeDead.png",
        "images/sprFireBallerIdle.png" => "images/sprFireBallerDead.png",
        "images/sprSuperFireBallerIdle.png" => "images/sprSuperFireBallerDead.png",
        "images/sprJockIdle.png" => "images/sprJockDead.png",
        "images/sprJungleFlyIdle.png" => "images/sprJungleFlyDead.png",
        "images/sprInvSpiderIdle.png" => "images/sprInvSpiderDead.png",
        "images/sprInvLaserCrystalIdle.png" => "images/sprInvLaserCrystalDead.png",
        "images/sprPopoFreakIdle.png" => "images/sprPopoFreakDead.png",
        "images/sprMSpawnIdle.png" => "images/sprMSpawnDead.png",
        "images/sprFrogQueenIdle.png" => "images/sprFrogQueenDead.png",
        "images/sprSniperIdle.png" => "images/sprSniperDead.png",
        "images/sprCrabIdle.png" => "images/sprCrabDead.png",
        "images/sprSpiderIdle.png" => "images/sprSpiderDead.png",
        "images/sprNecromancerIdle.png" => "images/sprNecromancerDead.png",
        "images/sprExploderIdle.png" => "images/sprExploderDead.png",
        "images/sprLaserCrystalIdle.png" => "images/sprLaserCrystalDead.png",
        "images/sprMimicIdle.png" => "images/sprMimicDead.png",
        "images/sprSuperMimicIdle.png" => "images/sprSuperMimicDead.png",
        "images/sprWepMimicIdle.png" => "images/sprWepMimicDead.png",
        "images/sprScrapBossIdle.png" => "images/sprScrapBossDead.png",
        "images/sprLilHunter.png" => "images/sprLilHunterDead.png",
        "images/sprLilHunterIdle.png" => "images/sprLilHunterDead.png",
        "images/sprHyperCrystalIdle.png" => "images/sprHyperCrystalDead.png",
        "images/sprGruntIdle.png" => "images/sprGruntDead.png",
        "images/sprShielderIdle.png" => "images/sprShielderDead.png",
        "images/sprEliteGruntIdle.png" => "images/sprEliteGruntDead.png",
        "images/sprEliteShielderIdle.png" => "images/sprEliteShielderDead.png",
        "images/sprEliteInspectorIdle.png" => "images/sprEliteInspectorDead.png",
        "images/sprInspectorIdle.png" => "images/sprInspectorDead.png",
        "images/sprFrogEgg.png" => "images/sprFrogEggDead.png",
        "images/sprCrystalProp.png" => "images/sprCrystalPropDead.png",
        "images/sprTechnoMancer.png" => "images/sprTechnoMancerDead.png",
        "images/sprVanDrive.png" => "images/sprVanDead.png",
        "images/sprSpookyBanditIdle.png" => "images/sprSpookyBanditDead.png",
        "images/sprYVBossIdle.png" => "images/sprYVBossDead.png",
        "images/sprCrownGuardianIdle.png" => "images/sprCrownGuardianDead.png",
        "images/sprIceFlowerIdle.png" => "images/sprIceFlowerDead.png",
        "images/sprSuperFrogIdle.png" => "images/sprSuperFrogDead.png",
        "images/sprRadMaggotIdle.png" => "images/sprRadMaggotDead.png",
        "images/sprLastIdle.png" => "images/sprLastDeath.png",
        "images/sprNothing2Idle.png" => "images/sprNothing2Death.png",
        "images/sprEnemyHorrorIdle.png" => "images/sprEnemyHorrorDead.png",
        "images/sprFiredMaggot.png" => "images/sprMaggotDead.png",
        "images/sprRadMaggot.png" => "images/sprRadMaggotDead.png",
        _ => idle,
    }
}
