(define (problem picture_388)
    (:domain ariac)
    
    (:objects
table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery blue_battery red_pump green_regulator green_regulator_1 red_battery - part
    )
    
    (:init
(robot_at table)
        (handempty)        
        (clear green_battery)
        (clear blue_battery)
        (on green_regulator_1 red_pump)
        (clear green_regulator)
        (clear green_regulator_1)
        (clear red_battery)
        (part_at red_pump table)
        (part_at green_battery table)
        (part_at green_regulator regulator_placement)
        (part_at blue_battery battery_placement)
        (part_at red_battery table)
    )
    
    (:goal
(and
            (part_at red_pump pump_placement)
        )
    )
)